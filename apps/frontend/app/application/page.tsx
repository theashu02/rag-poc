import React from "react";
import { ChatInterface } from "../components/chatUI/ChatInterface";
import { Sidebar } from "../components/common/Sidebar";

export default function page() {
  return (
    <>
      <div className="flex h-screen w-screen bg-background">
        <Sidebar />
        <ChatInterface />
      </div>
    </>
  );
}
