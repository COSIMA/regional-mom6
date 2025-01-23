# Docker Image & Github Testing (For contributors)

regional-mom6 uses a docker image in github actions for holding large data. It wasn't directly being used, but for downloading the curvilinear grid for testing, we are using it. This document is a list of helpful commands to work on it.

The link to the image is at: 
[https://github.com/COSIMA/regional-mom6/pkgs/container/regional-test-env](https://github.com/COSIMA/regional-mom6/pkgs/container/regional-test-env)

For local development of the image to add data to it for testing, first pull it. 
```docker pull ghcr.io/cosima/regional-test-env:updated```

Then to do testing of the image, we cd into our cloned version of regional-mom6, and run this command. It mounts our code in the /workspace directory.:
```docker run -it --rm \ -v $(pwd):/workspace \ -w /workspace \ ghcr.io/cosima/regional-test-env:updated \ /bin/bash```

The -it flag is for shell access, and the workspace stuff is to get our local code in the container.
You have to download conda, python, pip, and all that business to properly run the tests.

Getting to adding the data, you should create a folder and add both the data you want to add and a file simple called "Dockerfile". In Dockerfile, we'll get the original image, then copy the data we need to the data folder.

```
# Use the base image
FROM ghcr.io/cosima/regional-test-env:<tag>

# Copy your local file into the /data directory in the container
COPY <file> /data/<file>
```

Then, we need to build the image, tag it, and push it

```
docker build -t my-custom-image . # IN THE DIRECTORY WITH THE DOCKERFILE
docker tag my-custom-image ghcr.io/cosima/regional-test-env:<new_tag>
docker push ghcr.io/cosima/regional-test-env:<new_tag>
```
