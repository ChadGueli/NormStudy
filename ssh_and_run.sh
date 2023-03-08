#! /bin/sh

TEST=false
while getopts ":t" flag; do
  case "$flag" in
    t) TEST=true;;
    ?) echo "t indicates the experiment should be run in test mode. ./ssh_and_run.sh [-t] ip"
       exit 1 ;;
  esac
done

shift $((OPTIND-1))
IP=$@

if [ -z "$IP" ] ; then
  echo "Error: Need an ip address to ssh into as the first arg."
  exit 1
fi

AWS_CRED=`cat "${iCloud}"/Papers/WW/NormStudy/aws_cred.csv`

ssh -T -i ~/.ssh/LambdaCloudSSH.pem ubuntu@$IP << EOL

  echo "Running container on Lambda Cloud!"
  
  # volumes created in case of an aws issue
  sudo docker volume create data-in
  sudo docker volume create data-out

  if [[ $TEST = true ]] ; then
    echo "testing"
    sudo docker run -a stdout --name "experiment" --gpus "all"\
      --mount source=data-in,target=/data/in \
      --mount source=data-out,target=/data/out \
      chadgueli/norm-study:latest \
        python experiment.py -t --aws "$AWS_CRED"
  else
    echo "not testing"
    sudo docker run -d --name "experiment" --gpus "all"\
      --mount source=data-in,target=/data/in \
      --mount source=data-out,target=/data/out \
      chadgueli/norm-study:latest \
        python experiment.py --aws "$AWS_CRED"
  fi

EOL

echo "Back to your Mac"
