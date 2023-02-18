APP_DIR=
LIB_DIR=nebula_src
TEST_DIR=tests

analyze:
	radon cc ${APP_DIR} ${LIB_DIR} -s -n B
	bandit -r ${APP_DIR} ${LIB_DIR}
	isort --check ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	black --check ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	flake8 ${APP_DIR} ${LIB_DIR}

format:
	isort ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	black ${APP_DIR} ${LIB_DIR} ${TEST_DIR}
	flake8 ${APP_DIR} ${LIB_DIR} ${TEST_DIR}

run:
	echo "Not Implemented"