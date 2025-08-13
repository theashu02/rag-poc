import express from "express";

const app = express();

app.get("/api/v1/health", (req, res) => {
  res.status(200).json({ status: "Endpoints is working!" });
});

app.listen(5000);
