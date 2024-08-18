.PHONY: all create_environment create_quarto_book install_quarto install_ipykernel clean

all: create_environment create_quarto_book install_ipykernel

create_environment:
	# Create virtual environment
	python -m venv ndvi_env
	# Activate venv and install requirements
	. ndvi_env/bin/activate && pip install -r requirements.txt

create_quarto_book: create_environment
	# Activate venv and render Quarto book
	. ndvi_env/bin/activate && quarto render

install_quarto:
	wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.55/quarto-1.5.55-linux-amd64.deb
	sudo dpkg -i quarto-1.5.55-linux-amd64.deb

install_ipykernel: create_environment
	# Activate venv and install ipykernel
	. ndvi_env/bin/activate && pip install ipykernel
	. ndvi_env/bin/activate && python -m ipykernel install --user --name=ndvi_env_kernel

clean:
	rm -rf ndvi_env
	rm -f quarto-1.5.55-linux-amd64.deb
