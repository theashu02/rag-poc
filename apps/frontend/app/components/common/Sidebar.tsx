"use client"

import { useState } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { LogOut, Menu, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { signOut, useSession } from "next-auth/react"
import { useRouter } from "next/navigation"
import { ModeToggle } from "@/components/ThemeFiles/ThemeToggle"
import { useDispatch } from "react-redux"
import { clearUser } from "@/store/slices/UserStoreSlice"
import type { AppDispatch } from "@/store/store"

export function Sidebar() {
  const router = useRouter()
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const { data: session } = useSession()
  const dispatch = useDispatch<AppDispatch>()

  const handleMouseEnter = () => setIsCollapsed(false)
  const handleMouseLeave = () => setIsCollapsed(true)

  const toggleMobile = () => setIsMobileOpen(!isMobileOpen)

  const handleSignOut = () => {
    dispatch(clearUser())
    localStorage.removeItem("user-storage")
    signOut({ callbackUrl: "/" })
  }

  const HandleDoucuments = () => {
    router.push("/documents")
  }
  const handleMenu = () => {
    router.push("/")
  }
  const HandleHome = () => {
    router.push("/")
  }

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "fixed top-4 left-4 z-[60] md:hidden bg-sidebar text-sidebar-foreground hover:bg-sidebar-accent transition-all duration-200",
          isMobileOpen && "left-[272px]", // Move button when sidebar is open
        )}
        onClick={toggleMobile}
      >
        {isMobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {isMobileOpen && <div className="fixed inset-0 bg-black/50 z-[45] md:hidden" onClick={toggleMobile} />}

      <div
        className={cn(
          "fixed left-0 top-0 z-[50] h-screen bg-studio-bg border-r border-sidebar-border transition-all duration-300 ease-in-out",
          "md:relative md:z-auto",
          isCollapsed ? "w-16" : "w-64",
          isMobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        )}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
          {isCollapsed ? (
            <Button
              onClick={handleMenu}
              variant="ghost"
              className="w-full p-2 hover:cursor-pointer hover:bg-sidebar-accent flex items-center justify-center"
            >
              <Menu className="h-5 w-5 text-sidebar-foreground" />
            </Button>
          ) : (
            <Button
              onClick={handleMenu}
              variant="ghost"
              className="text-3xl font-bold hover:cursor-pointer dark:text-amber-100 hover:bg-transparent dark:hover:bg-transparent"
            >
              SimplX
            </Button>
          )}
        </div>

        {/* Content Area */}
        <div className="flex flex-col">
          {/* User Profile Section */}
          <div className="flex p-4 border-b border-sidebar-border items-center justify-center">
            <div className="flex items-center gap-3">
              <Avatar className="h-10 w-10 flex-shrink-0">
                <AvatarImage src={session?.user.image || "/placeholder.svg"} alt={session?.user.name || "User Name"} />
                <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground">
                  {session?.user.name
                    ?.split(" ")
                    .map((n) => n[0])
                    .join("")
                    .toUpperCase()}
                </AvatarFallback>
              </Avatar>
              {!isCollapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-sidebar-foreground truncate">{session?.user.name}</p>
                  <p className="truncate text-xs text-muted-foreground">{session?.user?.email || ""}</p>
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
                    onClick={HandleHome}
                    variant="ghost"
                    className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
                  >
                    Home
                  </Button>
                  <Button
                    onClick={HandleDoucuments}
                    variant="ghost"
                    className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
                  >
                    Documents
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
            <ModeToggle isCollapsed={isCollapsed} />
            <Button
              variant="ghost"
              onClick={handleSignOut}
              className={cn(
                "text-destructive hover:bg-[#F4F4F4] hover:text-gray-700 transition-colors dark:hover:text-amber-100",
                isCollapsed ? "w-full p-2" : "w-full justify-start",
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
