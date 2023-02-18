import typing
from concurrent import futures

import grpc
import keras
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[1]))  # isort:skip

from app import text_generator_pb2_grpc, text_generator_pb2
# from app.services import model

TEXT_GENERATOR: typing.Optional[keras.Model] = None


class TextGeneratorServicer(text_generator_pb2_grpc.TextGeneratorServicer):
    def SendPrompt(self, request: text_generator_pb2.TextPromptRequest, context) -> text_generator_pb2.TextPromptResponse:
        print(request.prompt)
        print('sending prompt back')
        return text_generator_pb2.TextPromptResponse(
            message='hello world from kevin'
        )


def main():
    # global TEXT_GENERATOR
    # TEXT_GENERATOR = model.load_model()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    text_generator_pb2_grpc.add_TextGeneratorServicer_to_server(
        servicer=TextGeneratorServicer(), server=server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
