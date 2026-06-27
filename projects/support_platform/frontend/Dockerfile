FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build the app (if we wanted to serve static, but here we just run dev for ease in a local container)
# RUN npm run build

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
