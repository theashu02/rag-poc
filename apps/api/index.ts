import express from "express";
import cors from "cors"

const app = express();
const PORT = process.env.PORT || 5000

app.use(cors({origin: "http://localhost:3000"}))

app.get("/api/v1/health", (req, res) => {
  res.status(200).json({ status: "Endpoints is working from the api folder!" });
});

app.listen(PORT);

console.log(`The api server is listening on the port ${PORT}`)
