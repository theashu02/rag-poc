"use client";
import React from "react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { HealthCheckApi } from "@/lib/ApiStore/ChatApi";

export default function HealthCheck() {
  const [health, setHealth] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await HealthCheckApi()
      setHealth(res?.data?.status ?? "OK");
    } catch (e: any) {
      setError(e?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <>
      <div className="absolute left-6 top-4 z-50 gap-3">
        <Button variant="secondary" onClick={checkHealth} disabled={loading}>
          {loading ? "Checking..." : "Check API Health"}{" "}
        </Button>
        {health && <p className="text-green-600">Health: {health}</p>}
        {error && <p className="text-red-600">Error: {error}</p>}{" "}
      </div>
    </>
  );
}
