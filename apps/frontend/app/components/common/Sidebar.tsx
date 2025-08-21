"use client"

import { useEffect, useState } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, LogOut, Menu } from 'lucide-react'
import { cn } from "@/lib/utils"
import { signOut, useSession } from "next-auth/react";
import { useRouter } from "next/navigation"
import { ModeToggle } from "@/components/ThemeToggle"
import { useUserStore } from "@/store/useUserStore"

export function Sidebar() {
  const router = useRouter();
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const { data: session } = useSession();
  const clearUser = useUserStore((s) => s.clearUser);

  const toggleCollapse = () => setIsCollapsed(!isCollapsed)
  const toggleMobile = () => setIsMobileOpen(!isMobileOpen)

  const handleSignOut = () => {
    clearUser();
    localStorage.removeItem('user-storage')
    signOut({ callbackUrl: "/" });
  }

  useEffect(()=>{
    if(!session){
      router.push("/")
    }
  }, [session, router])


  return (
    <>
      {/* Mobile Menu Button */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden bg-sidebar text-sidebar-foreground hover:bg-sidebar-accent"
        onClick={toggleMobile}
      >
        <Menu className="h-5 w-5" />
      </Button>

      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={toggleMobile}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          "fixed left-0 top-0 z-40 h-screen bg-sidebar border-r border-sidebar-border transition-all duration-300 ease-in-out",
          "md:relative md:z-auto",
          isCollapsed ? "w-16" : "w-64",
          isMobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        )}
      >
        {/* Header with Toggle Button */}
        <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
          {!isCollapsed && (
            <h2 className="text-lg font-semibold text-sidebar-foreground">
              Menu
            </h2>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleCollapse}
            className="hidden md:flex text-sidebar-foreground hover:bg-sidebar-accent ml-auto"
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Content Area */}
        <div className="flex flex-col">
          {/* User Profile Section */}
          <div className="flex p-4 border-b border-sidebar-border items-center justify-center">
            <div className="flex items-center gap-3">
              <Avatar className="h-10 w-10 flex-shrink-0">
                <AvatarImage src={session?.user.image || "/placeholder.svg"} alt={session?.user.name || "User Name"} />
                <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground">
                  {session?.user.name?.split(' ').map(n => n[0]).join('').toUpperCase()}
                </AvatarFallback>
              </Avatar>
              {!isCollapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-sidebar-foreground truncate">
                    {session?.user.name}
                  </p>
                  <p className="text-xs text-sidebar-foreground/70 truncate">
                    {session?.user.email}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Navigation Area */}
          <div className="flex-1 p-4">
            {!isCollapsed && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-sidebar-foreground/70 uppercase tracking-wider">
                  Navigation
                </div>
                <div className="space-y-1">
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
                  >
                    Dashboard
                  </Button>
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
                  >
                    Settings
                  </Button>
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
                  >
                    Profile
                  </Button>
                </div>
              </div>
            )}
          </div>
      
          {/* Sign Out Button */}
          <div className="absolute bottom-0 p-4 border-t border-sidebar-border w-full flex flex-col justify-center items-center gap-3">
            <ModeToggle isCollapsed={isCollapsed}/>
            <Button
              variant="ghost"
              onClick={handleSignOut}
              className={cn(
                "text-destructive hover:bg-[#F4F4F4] hover:text-destructive-foreground transition-colors",
                isCollapsed ? "w-full p-2" : "w-full justify-start"
              )}
            >
              <LogOut className="h-4 w-4" />
              {!isCollapsed && <span className="ml-2">Sign Out</span>}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}