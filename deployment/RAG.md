# ✨ Retrieval Augmented Generation (RAG)

<details close>
<summary><b><font size="5">Basic</font></b></summary>
<br>
<a href="https://colab.research.google.com/drive/1peyWgBM80OwY3SOy-uN8ZcjR2ME_Qhdq?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>
</details>

<br>

<details open>
<summary><b><font size="5">Advanced</font></b></summary>
<br></br>
<img src="https://miro.medium.com/v2/resize:fit:1400/0*q9OFsbhJZ44E47bD.png" alt="Flowise" width="800" height="300">
<br>
<br>
Flowise is an open source low-code tool for developers to build customized LLM orchestration flow & AI agents. You can find more <a href="https://docs.flowiseai.com">details</a>.
<h4>Step 1: Establish The Necessary Network</h4>

```bash
docker network create --driver bridge retrieval-augmented-generation
```

<h4>Step 2: Prepare the Embeddings and LLM service.</h4>

Prepare the embeddings and model YAML file at the current working directory. These file can be found in the `deployment/LocalAI` which is `bge-m3.yaml` and `LLaMa3-8b-WangchanX-sft-Demo.yaml`. Then run the following command for create service

```bash
docker build -t localai -f deployment/LocalAi/Dockerfile.LocalAi .
docker run -d -it --network retrieval-augmented-generation --gpus '"device=0"' -p 8080:8080 --name localai-service localai
```

<h4>Step 3: Prepare Retriever service.</h4>

Run the following command for create service

```bash
 docker run -d --network retrieval-augmented-generation --name chroma-db -p 8000:8000 chromadb/chroma
```

Run `upload.py` that can find in `Chroma/upload.py` for upload embeddings documents to retriever db

```bash
python upload.py
```

<h4>Step 4: Prepare Analyze Chatflow service.</h4>

Run the following command for create chatflow database service

```bash
docker run -d --name langfuse-db --network retrieval-augmented-generation -p 5432:5432 -e POSTGRES_USER=langfuse -e POSTGRES_PASSWORD=WangchanX -e POSTGRES_DB=postgres postgres:14
```

Run the following command for create chatflow service

```bash
docker run -it --network retrieval-augmented-generation -e DATABASE_HOST=langfuse-db -e DATABASE_USERNAME=langfuse -e DATABASE_PASSWORD=WangchanX -e DATABASE_NAME=postgres -e NEXTAUTH_URL=http://localhost:3000 -e NEXTAUTH_SECRET=mysecret -e SALT=mysalt -p 3000:3000 -d --name langfuse-service langfuse/langfuse
```

<h4>Step 5: Create Flowise service.</h4>

```bash
docker run -d -it --network retrieval-augmented-generation --name flowise-service -e PORT=4000 -e FLOWISE_USERNAME=admin -e FLOWISE_PASSWORD=admin -p 4000:4000 elestio/flowiseai
```

- Preview:

![Flowise](./img/flowise.gif)

</details>
