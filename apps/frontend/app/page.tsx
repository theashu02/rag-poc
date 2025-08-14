"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import api from "@/lib/axios";
// import Chat from "./components/Chat";
import { ChatInterface } from "./components/chatUI/ChatInterface";

export default function Home() {
	const [health, setHealth] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const checkHealth = async () => {
		setLoading(true);
		setError(null);
		try {
			const res = await api.get("/api/v1/health");
			setHealth(res.data?.status ?? "OK");
		} catch (e: any) {
			setError(e?.message ?? "Request failed");
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="font-sans grid items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20">
			<div className="flex flex-col items-center gap-4">
				<Button onClick={checkHealth} disabled={loading}>
					{loading ? "Checking..." : "Check API Health"}
				</Button>
				{health && <p className="text-green-600">Health: {health}</p>}
				{error && <p className="text-red-600">Error: {error}</p>}
			</div>
			<div className="flex flex-col items-center gap-4 w-5xl bg-red-300">
				{/* <Chat /> */}
				<ChatInterface />
			</div>
		</div>
	);
}