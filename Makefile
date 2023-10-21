install:
	pip install --upgrade pip &&\
		pip install pytest
		pip install black
		pip install pylint
		pip install -r requirements.txt

format:
	@find "$(PWD)" -name "*.py" -not -path "$(PWD)/.venv/*" -exec black {} +	

lint:
	pylint --output-format=colorized --disable=R,C *.py

run:
	python tune.py

tensorboard:
	tensorboard --logdir=tb_logs

monitor:
	nvtop

sweep:
	lightning run app sweep.py

make tune:
	make format run

make clean:
	rm -rf lightning_logs/* &&\
	rmdir lightning_logs &&\
	rm -rf tb_logs/* &&\
	rmdir tb_logs
