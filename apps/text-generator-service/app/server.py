import pathlib
import sys
import typing
from concurrent import futures

import grpc
import keras

sys.path.append(str(pathlib.Path(__file__).parents[1]))  # isort:skip

from app import text_generator_pb2, text_generator_pb2_grpc  # noqa: E402
from app.services import model  # noqa: E402

TEXT_GENERATOR: typing.Optional[keras.Model] = None


class TextGeneratorServicer(text_generator_pb2_grpc.TextGeneratorServicer):
    def SendPrompt(self, request: text_generator_pb2.TextPromptRequest, context) -> text_generator_pb2.TextPromptResponse:
        global TEXT_GENERATOR
        print(request.prompt)
        response = TEXT_GENERATOR.predict()
        TEXT_GENERATOR.predict

        response = 'manual kevin'
        print('sending prompt back ', response)
        return text_generator_pb2.TextPromptResponse(message=response)


def main():
    global TEXT_GENERATOR
    if TEXT_GENERATOR is None:
        TEXT_GENERATOR = model.load_model()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    text_generator_pb2_grpc.add_TextGeneratorServicer_to_server(servicer=TextGeneratorServicer(), server=server)
    server.add_insecure_port('[::]:50051')
    print('Starting gRPC server...')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
