from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    """Bound incoming JSON/multipart bodies before parsing, including chunked uploads."""

    def __init__(self, app, max_bytes=65_536):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                response = JSONResponse(
                    {
                        "error": {
                            "code": "input_too_large",
                            "message": "Request body exceeds 64 KiB.",
                        }
                    },
                    status_code=413,
                )
                return await response(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
