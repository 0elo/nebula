# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from app import text_generator_pb2 as app_dot_text__generator__pb2


class TextGeneratorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendPrompt = channel.unary_unary(
            '/text_generator.TextGenerator/SendPrompt',
            request_serializer=app_dot_text__generator__pb2.TextPromptRequest.SerializeToString,
            response_deserializer=app_dot_text__generator__pb2.TextPromptResponse.FromString,
        )


class TextGeneratorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendPrompt(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TextGeneratorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'SendPrompt': grpc.unary_unary_rpc_method_handler(
            servicer.SendPrompt,
            request_deserializer=app_dot_text__generator__pb2.TextPromptRequest.FromString,
            response_serializer=app_dot_text__generator__pb2.TextPromptResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler('text_generator.TextGenerator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class TextGenerator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendPrompt(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/text_generator.TextGenerator/SendPrompt',
            app_dot_text__generator__pb2.TextPromptRequest.SerializeToString,
            app_dot_text__generator__pb2.TextPromptResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
