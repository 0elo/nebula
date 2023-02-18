APP_DIR=app
LIB_DIR=
TEST_DIR=tests

analyze:
	radon cc ${APP_DIR} ${LIB_DIR} -s -n B
	bandit -r ${APP_DIR} ${LIB_DIR}
	isort --check ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	black --check ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	flake8 ${APP_DIR} ${LIB_DIR}

create_proto:
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. app/text_generator.proto

format:
	isort ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	black ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	flake8 ${APP_DIR} ${LIB_DIR} ${TEST_DIR}

run:
	python app/server.py

request:
	python app/client.py