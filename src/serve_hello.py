from fastapi import FastAPI
from ray import serve
import os

# When the env var is updated, users see new return value.
msg = os.getenv("SERVE_RESPONSE_MESSAGE", "Hello world!")

app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class HelloWorld:
    @app.get("/")
    def hello(self):
        return msg

entrypoint = HelloWorld.bind()