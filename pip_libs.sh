#!/usr/bin/env bash
# author: David Zhang

# this script is used to install all dependent libs

function usage() {
	cat <<EOF
Usage: bash pip_libs.sh [--mirror=<mirror-url> | -M]
Options:
  --mirror=<mirror-url>:
    will use a mirror you specify to install libs
  -M:
    just use a default mirror to install libs
    default: https://mirrors.aliyun.com/pypi/simple
if any options be not specified, it will not use a mirror to install libs existed in the file requirements.txt.
EOF
}

mirror=
libs_file=requirements.txt

while (( $# > 0 )); do
  case "$1" in
    --mirror=?*)
    mirror=$(echo "$1" | sed 's/--mirror=//' | sed "s/['\"]//g")
    shift
    ;;
    -M)
      mirror='https://mirrors.aliyun.com/pypi/simple'
      shift
      ;;
    -h)
      usage
      exit 0
      ;;
    *)
      echo 'Error: options you input are illeagal, more details, see -h' >&2
      exit 1
  esac
done

# removes the notes
libs=$(cat < $libs_file | grep -Ev '^\s*#|^\s*$' | sed 's/#.*//' | xargs)
echo -e "[INFO] these libs will be installed:\n$libs"

if [[ -z $mirror ]]; then
  echo "[INFO] pip is installing without a mirror."
  pip install $libs
else
  echo "[INFO] pip is installing with the mirror $mirror."
  pip install $libs -i $mirror
fi

exit $?