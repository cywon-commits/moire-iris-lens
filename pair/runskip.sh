# usage: runskip.sh LAM mode d V0 alpha orient
export PAIR_LAM=$1; shift
tag="$1_d$(python3 -c "print(float($2))")_V$3_a$4_$5"
[ "$PAIR_LAM" != "1.0" ] && tag="${tag}_lam$PAIR_LAM"
[ -f "$tag.json" ] && { echo "skip $tag"; exit 0; }
python3 pair.py "$@"
