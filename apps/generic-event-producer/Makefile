CARGO := cargo
CLIPPY := cargo clipy -- -D warnings
RUSTFMT := cargo fmt

.PHONY: build lint format clean

build:
	${CARGO} build

lint:
	$(CLIPPY)

format:
	$(RUSTFMT)

clean:
	$(CARGO) clean

run:
	$(CARGO) run
