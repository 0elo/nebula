import logging
import random
import sys
import pathlib
import grpc

sys.path.append(str(pathlib.Path(__file__).parents[1]))  # isort:skip

import text_generator_pb2
import text_generator_pb2_grpc



def make_text_generator_request():
    return text_generator_pb2.TextPromptRequest(prompt='prompt from client')


def send_text_generator_request(stub):
    request = make_text_generator_request()
    return stub.SendPrompt(request)


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = text_generator_pb2_grpc.TextGeneratorStub(channel)
        print('Sending request.')
        response = send_text_generator_request(stub)
        print(response.message)
        print('Request done.')


if __name__ == '__main__':
    logging.basicConfig()
    run()
