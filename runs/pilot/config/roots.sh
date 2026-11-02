for img in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep format-code); do
  r=$(docker run --rm --entrypoint bash $img -c 'cd /testbed && git rev-list --max-parents=0 HEAD | head -1; git rev-list --count HEAD' 2>/dev/null | tr '\n' ' ')
  echo "${img##*:} $r"
done
