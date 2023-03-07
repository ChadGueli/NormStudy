#! /bin/sh

TEST=false
while getopts ":t" flag; do
  case "$flag" in
    t) TEST=true;;
    ?) echo "t indicates the experiment should be run in test mode. ./_ssh_and_run.sh [-t] ip"
       exit 1 ;;
  esac
done

shift $((OPTIND-1))
IP=$@

if [ -z "$IP" ] ; then
  echo "Error: Need an ip address to ssh into as the first arg."
  exit 1
fi

cd "${iCloud}"/Papers/WW/NormStudy
AWS_CRED=`cat aws_cred.csv`
GITHUB_TOKEN=`cat ghtoken.txt`

ssh -T -i ~/.ssh/LambdaCloudSSH.pem ubuntu@$IP << EOL

  echo "Downloading git repo"
  git clone -b test https://$GITHUB_TOKEN@github.com/ChadGueli/NormStudy.git
  cd NormStudy/code

  echo "Setting up environment"
  aws configure import --csv "$AWS_CRED"
  # it is faster to update everytime than to pull the massive docker container
  pip install -U --quiet boto3[crt] optuna pytorch-lightning \
    numexpr torchvision

  mkdir -p data/out
  mkdir data/in
  
  if [[ $TEST = true ]] ; then
    echo "testing"
    python experiment.py -t
  else
    echo "not testing"
    python experiment.py
  fi

EOL