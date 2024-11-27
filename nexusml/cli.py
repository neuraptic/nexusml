import subprocess
import time

import click


@click.command()
def run_server():
    # Start celery worker
    worker_process = subprocess.Popen(['celery', '-A', 'nexusml.api.make_celery', 'worker'])
    time.sleep(10)

    # Start celery beat
    beat_process = subprocess.Popen(['celery', '-A', 'nexusml.api.make_celery', 'beat'])
    time.sleep(2)

    # Start flask app
    flask_process = subprocess.Popen(['flask', '--app', 'nexusml.api', 'run'])

    # Wait for all processes to complete
    worker_process.wait()
    beat_process.wait()
    flask_process.wait()


if __name__ == '__main__':
    run_server()
