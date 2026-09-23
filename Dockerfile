FROM --platform=linux/arm64 vllm/vllm-openai@sha256:865784ba59b46e3e1370df8158c1bbff8769c3ab88d63551d3f25eabc1a61c45
LABEL org.opencontainers.image.title="MiMo 2.6 on DGX Spark" \
      org.opencontainers.image.source="https://github.com/Aevonix/mimo-2.6-dgx-spark"
COPY manifest.json LICENSE NOTICE.md /opt/mimo-spark/
COPY overlay /opt/mimo-spark/overlay
COPY artifacts /opt/mimo-spark/artifacts
COPY scripts/install_overlays.py /opt/mimo-spark/scripts/install_overlays.py
COPY config/trace-disabled.json /opt/mimo-spark/trace-disabled.json
RUN python /opt/mimo-spark/scripts/install_overlays.py --site-packages /usr/local/lib/python3.12/dist-packages
ENTRYPOINT ["vllm"]
