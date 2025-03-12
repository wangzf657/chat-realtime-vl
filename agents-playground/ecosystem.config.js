// ecosystem.config.js
module.exports = {
    apps: [
      {
        name: "agents-app",
        script: "npm", // 或者 "yarn"
        args: "start", // 或者 "start"
        instances: "max", // 或者指定具体的进程数，例如 2
        autorestart: true, // 自动重启
        watch: false, // 是否监视文件变化并自动重启（开发环境可以设置为 true）
        max_memory_restart: "1G", // 如果内存使用超过 1GB，则自动重启
        env: {
          NODE_ENV: "development", // 开发环境
        },
        env_production: {
          NODE_ENV: "production",
          PORT: 3000, 
        },
      },
    ],
  };