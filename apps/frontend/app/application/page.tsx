'use client'

import React, { useEffect, useState } from "react";
import { ChatInterface } from "../components/chatUI/ChatInterface";
import { Sidebar } from "../components/common/Sidebar";
import { StudioSidebar } from "../components/common/StudioSidebar";
import { modernToast } from "@/lib/toast"
import { useSession } from "next-auth/react"

export default function page() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const { data: session } = useSession();

  useEffect(() => {
    if (session && !localStorage.getItem("app_toast_shown")) {
      modernToast.success(`Welcome back, ${session.user.name}! Let's make today productive.`);
      localStorage.setItem("app_toast_shown", "true");
    }
  }, [session]);

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