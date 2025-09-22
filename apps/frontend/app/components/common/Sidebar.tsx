"use client";

import { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { LogOut, Menu, X, Home, Compass, Grid3X3 } from "lucide-react";
import { cn } from "@/lib/utils";
import { signOut, useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { ModeToggle } from "@/components/ThemeFiles/ThemeToggle";
import { useDispatch } from "react-redux";
import { clearUser } from "@/store/slices/UserStoreSlice";
import type { AppDispatch } from "@/store/store";
import { modernToast } from "@/lib/toast";
import Image from "next/image";

export function Sidebar() {
  const router = useRouter();
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const { data: session } = useSession();
  const dispatch = useDispatch<AppDispatch>();
  const ImageURL = "https://media.licdn.com/dms/image/v2/D560BAQEikpv0408BzQ/company-logo_100_100/company-logo_100_100/0/1730929317508/observeai_logo?e=1761177600&v=beta&t=ixsAbKcB9ULvqjjRVrvO66kQU-KqG_hRaFlgfQkxEes";

  console.log("current logged in user", session);

  const handleMouseEnter = () => setIsCollapsed(false);
  const handleMouseLeave = () => setIsCollapsed(true);
  const toggleMobile = () => setIsMobileOpen(!isMobileOpen);

  const handleSignOut = () => {
    modernToast.success(
      `Thanks for visiting, ${session?.user?.name || "User"}!`
    );
    setTimeout(() => {
      dispatch(clearUser());
      localStorage.removeItem("user-storage");
      localStorage.removeItem("app_toast_shown");
      signOut({ callbackUrl: "/" });
    }, 1200);
  };

  const handleMenu = () => {
    router.push("/application");
  };

  const HandleHome = () => {
    router.push("/");
  };

  const navigationItems = [
    { icon: Home, label: "Home", onClick: HandleHome },
    { icon: Compass, label: "Discover", onClick: () => {} },
    { icon: Grid3X3, label: "Spaces", onClick: () => {} },
  ];

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "fixed top-4 left-4 z-[60] md:hidden bg-sidebar text-sidebar-foreground hover:bg-sidebar-accent transition-all duration-200",
          isMobileOpen && "left-[272px]"
        )}
        onClick={toggleMobile}
      >
        {isMobileOpen ? (
          <X className="h-5 w-5" />
        ) : (
          <Menu className="h-5 w-5" />
        )}
      </Button>

      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-sidebar text-sidebar-foreground hover:bg-sidebar-accent z-[45] md:hidden"
          onClick={toggleMobile}
        />
      )}

      <div
        className={cn(
          "fixed left-0 top-0 z-[50] h-screen bg-studio-bg border-r border-sidebar-border transition-all duration-300 ease-in-out flex flex-col",
          "md:relative md:z-auto",
          isCollapsed ? "w-20" : "w-64",
          isMobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        )}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* Header */}
        <div className="flex items-center py-4 justify-between border-b border-sidebar-border mx-auto">
          {isCollapsed ? (
            <Button
              onClick={handleMenu}
              variant="ghost"
              className="w-full p-2 hover:cursor-pointer hover:bg-sidebar-accent flex items-center justify-center"
            >
              <Image src={ImageURL} alt="logo" className="rounded-2xl" width={40} height={40} />
            </Button>
          ) : (
            <Button
              onClick={handleMenu}
              variant="ghost"
              className="text-3xl font-bold hover:cursor-pointer dark:text-amber-100 hover:bg-transparent dark:hover:bg-transparent"
            >
              SimpLX
            </Button>
          )}
        </div>

        <div className="flex-1 py-4 space-y-3 mx-auto w-full">
          {navigationItems.map((item, index) => (
            <Button
              key={index}
              onClick={item.onClick}
              variant="ghost"
              className={cn(
                "w-full h-12 justify-start rounded-sm transition-colors flex items-center group mx-auto cursor-pointer text-sidebar-foreground hover:bg-sidebar-accent"
              )}
            >
              <div
                className={cn(
                  "w-10 h-10 flex flex-col items-center justify-center transition-colors"
                )}
              >
                <item.icon style={{ width: "60%", height: "60%" }} />
                {isCollapsed && (
                  <span className="text-[10px]">{item.label}</span>
                )}
              </div>

              {!isCollapsed && (
                <span className="ml-3 text-sm flex-1 text-left font-medium">
                  {item.label}
                </span>
              )}
            </Button>
          ))}
        </div>

        {/* Bottom Items */}
        <div className="px-2 py-4 space-y-2">
          {/* User Profile */}
          <div className="flex items-center justify-center pt-2">
            <Avatar className="h-10 w-10 flex-shrink-0">
              <AvatarImage
                src={session?.user.image || "/placeholder.svg"}
                alt={session?.user.name || "User Name"}
              />
              <AvatarFallback className="bg-gray-600 text-white">
                {session?.user.name
                  ?.split(" ")
                  .map((n) => n[0])
                  .join("")
                  .toUpperCase()}
              </AvatarFallback>
            </Avatar>
            {!isCollapsed && (
              <div className="ml-3 flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 dark:text-gray-300 truncate">
                  {session?.user.name}
                </p>
                <p className="truncate text-xs text-gray-500">
                  {session?.user?.email || ""}
                </p>
              </div>
            )}
          </div>

          <div className="flex flex-col space-y-3 items-center justify-center pt-2">
            <ModeToggle isCollapsed={isCollapsed} />
            <Button
              variant="ghost"
              onClick={handleSignOut}
              className={cn(
                "w-full h-10 flex items-center rounded-full cursor-pointer text-destructive hover:bg-[#F4F4F4] hover:text-gray-700 transition-colors dark:hover:text-amber-100",
                isCollapsed ? "justify-center" : "justify-start px-3 border-2"
              )}
            >
              <LogOut className="h-5 w-5 text-stone-800 dark:text-red-500" />
              {!isCollapsed && (
                <span className="ml-3 text-sm flex-1 text-left">Sign Out</span>
              )}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}