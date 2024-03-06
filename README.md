This repository contains code for deploying trained AI-Model.

The detail of the AI-model is descripted in [this link](https://github.com/dokyung36d/2023-AICOSS)

I have deployed this project using DockerHub. So you can utilize my project in dockerhub by using below command.

<pre><code>
docker pull dokyung36d/aicoss:fastapi
</code></pre>

If you want to get AI-Prediciton, then send request with satellite image, then you can get the results showing whether object exists or not in the image.

the response's body format is JSON.