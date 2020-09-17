#!/usr/bin/env bash
# author: David Zhang

# this script is used to install all dependent libs

function usage() {
	cat <<EOF
Usage: bash pip_libs.sh [--mirror=<mirror-url>|-M] [-h|--help] [--pip=<pip-path>]
Options:
  --mirror=<mirror-url>:
    will use a mirror you specify to install libs
  -M:
    just use a default mirror to install libs
    default: https://mirrors.aliyun.com/pypi/simple
  --pip=<pip-path>:
    specifies the path pip installed or it in the sys path.
if any option is not specified, it will not use a mirror
  to install libs listed in the file 'requirements.txt'.
EOF
}

mirror=
libs_file=requirements.txt
pip=pip

while (( $# > 0 )); do
  case "$1" in
    --mirror=?*)
      mirror=$(echo "$1" | sed 's/--mirror=//' | sed "s/['\"]//g")
      shift
      ;;
    --pip=?*)
      pip=$(echo "$1" | sed 's/--pip=//' | sed "s/['\"]//g")
      shift
      ;;
    -M)
      mirror='https://mirrors.aliyun.com/pypi/simple'
      shift
      ;;
    -h|--help)
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
  "$pip" install $libs
else
  echo "[INFO] pip is installing with the mirror $mirror."
  "$pip" install $libs -i $mirror
fi

exit $?