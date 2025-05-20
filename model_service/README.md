
Using Makefile:
```bash
make build     # Build the containers
make up        # Start the containers
make up-d      # Start the containers in detached mode
make down      # Stop the containers
make clean     # Remove all containers, networks, and volumes
make logs      # View container logs
make restart   # Restart the containers
make ps        # Show running containers
make rebuild   # Rebuild and start containers
make rebuild-d # Rebuild and start containers in detached mode
```

## Development

1. Make changes to the code
2. Rebuild the containers:
   ```bash
   make rebuild
   ```
3. View logs:
   ```bash
   make logs
   ```

## Troubleshooting

1. If containers fail to start:
   ```bash
   make clean
   make rebuild-d
   ```

2. To check container status:
   ```bash
   make ps
   ```

3. To view logs:
   ```bash
   make logs
   ```

### Read docs

Swagger UI: http://127.0.0.1:8000/docs