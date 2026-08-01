.PHONY: binding notebooks pdfs artifacts test clean-generated

binding:
	./python/build_binding.sh

notebooks: binding
	PYTHONPATH=python python3 -W error scripts/generate_notebooks.py

pdfs: binding
	PYTHONPATH=python python3 -W error scripts/generate_pdfs.py

artifacts: notebooks pdfs

test: binding
	cargo fmt --all -- --check
	cargo clippy --all-targets --features python -- -D warnings
	cargo test --all-targets
	PYTHONPATH=python python3 -W error -m unittest discover -s python/tests -v
	python3 -W error -m compileall -q python scripts
	PYTHONPATH=python python3 -W error scripts/validate_notebooks.py

clean-generated:
	rm -f python/elspice_mna/_native.so
	rm -f artifacts/*.aux artifacts/*.log artifacts/*.out
