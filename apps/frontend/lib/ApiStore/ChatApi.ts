import api from "./axios";

export async function HealthCheckApi() {
  try {
    const res = await api.get("/api/v1/health");
    return res;
  } catch (e: any) {
    console.error("error in the api call: ", e);
  }
}
