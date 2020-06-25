#!/bin/bash

# How to use:
# ./build_push_dockerimage.sh --git_login mkolomeychenko --git_password yyy

# https://brianchildress.co/named-parameters-in-bash/
git_login=${git_login:-none}
git_password=${git_password:-none}

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi

    shift
done

docker build \
    --build-arg GIT_LOGIN=${git_login} \
    --build-arg GIT_PASSWORD=${git_password} \
    -t docker.deepsystems.io/supervisely/five/app_report . &&
    docker push docker.deepsystems.io/supervisely/five/app_report

#--no-cache \
