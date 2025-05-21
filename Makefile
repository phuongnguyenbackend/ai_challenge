.PHONY: build up down clean logs restart

# Build the containers
build:
	docker-compose build

# Start the containers
up:
	docker-compose up

# Start the containers in detached mode
up-d:
	docker-compose up -d

# Stop the containers
down:
	docker-compose down

# Remove all containers, networks, and volumes
clean:
	docker-compose down -v
	docker system prune -f

# View logs
logs:
	docker-compose logs -f

# Restart the containers
restart:
	docker-compose restart

# Show running containers
ps:
	docker-compose ps

# Rebuild and start containers
rebuild: build up

# Rebuild and start containers in detached mode
rebuild-d: build up-d

# Help command
help:
	@echo "Available commands:"
	@echo "  make build     - Build the containers"
	@echo "  make up        - Start the containers"
	@echo "  make up-d      - Start the containers in detached mode"
	@echo "  make down      - Stop the containers"
	@echo "  make clean     - Remove all containers, networks, and volumes"
	@echo "  make logs      - View container logs"
	@echo "  make restart   - Restart the containers"
	@echo "  make ps        - Show running containers"
	@echo "  make rebuild   - Rebuild and start containers"
	@echo "  make rebuild-d - Rebuild and start containers in detached mode" 