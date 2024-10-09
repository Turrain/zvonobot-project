FROM golang:1.23
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download && go mod verify
COPY . .
EXPOSE 9092
RUN go build -o main .
CMD ["./main"]