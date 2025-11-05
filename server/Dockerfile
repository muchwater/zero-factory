# Use the official Node.js image as the base image
FROM node:22

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to the working directory
COPY package.json ./
COPY pnpm-lock.yaml ./

# Install the application dependencies
RUN npm install -g pnpm
RUN pnpm install

# Copy the rest of the application files
COPY . .

# Generate Prisma Client
RUN pnpm prisma:generate

# Build the NestJS application
RUN pnpm run build

# Expose the application port
EXPOSE 3000

# Command to run migrations and start the application
CMD ["sh", "-c", "pnpm prisma:migrate:deploy && node dist/src/main.js"]
