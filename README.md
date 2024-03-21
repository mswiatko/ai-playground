# ai-playground

## Setup

1. Install Ollama
[linux example]
curl -fsSL https://ollama.com/install.sh | sh

2. Create virtual environment:

$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt

3. Download and run Ollama model:

$ ollama run llama2

$ ollama list -> will list downloaded models; models can be also imported

### Solving proxy issue:

$ vim /etc/systemd/system/ollama.service

- under [Service] add:
Environment="https_proxy=http://mycorporateproxy.local:8080"

- save the file

- reset ollama service:
$systemctl daemon-reload
$systemctl restart ollama.service
