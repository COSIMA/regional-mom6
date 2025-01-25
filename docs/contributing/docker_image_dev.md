# Docker Image & Github Testing (For contributors)

regional-mom6 uses a docker image in Github actions for holding large data. Here, we explain how contributors can use the docker image.

First things first install Docker by following [instructions at the docker docs](https://docs.docker.com/get-started/).

The docker image lives at: 
[https://github.com/COSIMA/regional-mom6/pkgs/container/regional-test-env](https://github.com/COSIMA/regional-mom6/pkgs/container/regional-test-env)

For local development of the image, e.g., to add data to it that will be used in the packages tests, first we need to pull it.

```bash
docker pull ghcr.io/cosima/regional-test-env:updated
```

Then to test the image, we go into the directory of our locally copy of regional-mom6, and run:

```
docker run -it --rm \ -v $(pwd):/workspace \ -w /workspace \ ghcr.io/cosima/regional-test-env:updated \ /bin/bash
```

The above command mounts the local copy of the package in the `/workspace` directory of the image.

The `-it` flag is for shell access; the workspace stuff is to get our local code in the container.
We need to download conda, python, pip, and all that business to properly run the tests.

To add data, we create a directory and add both the data we want and a file called `Dockerfile`.
Within `Dockerfile`, we'll get the original image, then copy the data we need to the data directory.

```
# Use the base image
FROM ghcr.io/cosima/regional-test-env:<tag>

# Copy your local file into the /data directory in the container
COPY <file> /data/<file>
```

Then, we need to build the image, tag it, and push it up.

```bash
docker build -t my-custom-image . # IN THE DIRECTORY WITH THE DOCKERFILE
docker tag my-custom-image ghcr.io/cosima/regional-test-env:<new_tag>
docker push ghcr.io/cosima/regional-test-env:<new_tag>
```
