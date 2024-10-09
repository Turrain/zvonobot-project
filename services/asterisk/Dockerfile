# Use a base image with necessary build tools
FROM debian:bullseye-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    wget \
    subversion \
    libxml2-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libedit-dev \
    uuid-dev \
    libjansson-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src

# Clone the Asterisk source code
RUN git clone https://github.com/asterisk/asterisk.git && \
    cd asterisk && \
    git checkout 18.10.1  # Specify the version you want to build

# Build Asterisk
RUN cd asterisk && \
    ./configure && \
    make && \
    make install && \
    make samples

# Copy configuration files
COPY ./config/extensions.conf /etc/asterisk/extensions.conf
COPY ./config/pjsip.conf /etc/asterisk/pjsip.conf


# Expose necessary ports for Asterisk
EXPOSE 5060/udp 5060/tcp 10000-20000/udp 5038/tcp 8088/tcp 9092

# Command to run Asterisk in the foreground
CMD ["asterisk", "-f"]