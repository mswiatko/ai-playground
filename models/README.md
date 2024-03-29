# AI models

## How to import models in Ollama

- download gguf file from model database (like Hugging Face)

- write modelfile for it

```
FROM file.gguf
```

Ther can be also additional information needed for model, like prompt, temperature, etc.

- create new model (named example)

```
$ ollama create example -f modelfile
```

- check if it is visiable

```
$ ollama list
```

Example output:

```
NAME                                    ID              SIZE    MODIFIED
all-minilm-l6:latest                    ac1f4ec31efa    21 MB   4 seconds ago
dolphin-mixtral:latest                  cfada4ba31c7    26 GB   7 days ago
llama2:latest                           78e26419b446    3.8 GB  7 days ago
```
