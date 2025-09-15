'use client'

import React, { useState } from "react";
import { ChatInterface } from "../components/chatUI/ChatInterface";
import { Sidebar } from "../components/common/Sidebar";
import { StudioSidebar } from "../components/common/StudioSidebar";

export default function page() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className="flex h-screen w-screen bg-background">
      <Sidebar />
      <ChatInterface />
      <StudioSidebar
        isCollapsed={sidebarCollapsed}
        onToggle={toggleSidebar}
      />
    </div>
  );
}