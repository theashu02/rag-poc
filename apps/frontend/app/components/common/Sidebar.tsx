// "use client"

// import { useState } from "react"
// import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
// import { Button } from "@/components/ui/button"
// import { LogOut, Menu, X } from "lucide-react"
// import { cn } from "@/lib/utils"
// import { signOut, useSession } from "next-auth/react"
// import { useRouter } from "next/navigation"
// import { ModeToggle } from "@/components/ThemeFiles/ThemeToggle"
// import { useDispatch } from "react-redux"
// import { clearUser } from "@/store/slices/UserStoreSlice"
// import type { AppDispatch } from "@/store/store"
// import { modernToast } from "@/lib/toast"

// export function Sidebar() {
//   const router = useRouter()
//   const [isCollapsed, setIsCollapsed] = useState(true)
//   const [isMobileOpen, setIsMobileOpen] = useState(false)
//   const { data: session } = useSession()
//   const dispatch = useDispatch<AppDispatch>()

//   console.log("current logged in user",session)

//   const handleMouseEnter = () => setIsCollapsed(false)
//   const handleMouseLeave = () => setIsCollapsed(true)

//   const toggleMobile = () => setIsMobileOpen(!isMobileOpen)

//   // const handleSignOut = () => {
//   //   dispatch(clearUser())
//   //   localStorage.removeItem("user-storage")
//   //   signOut({ callbackUrl: "/" })
//   //   modernToast.success(`Thanks for visiting, ${session?.user.name}!`)
//   // }
//   const handleSignOut = () => {
//     modernToast.success(`Thanks for visiting, ${session?.user?.name || "User"}!`)
//     setTimeout(() => {
//       dispatch(clearUser())
//       localStorage.removeItem("user-storage")
//       localStorage.removeItem("app_toast_shown")
//       signOut({ callbackUrl: "/" })
//     }, 1200)
//   }

//   const handleMenu = () => {
//     router.push("/application")
//   }

//   const HandleHome = () => {
//     router.push("/")
//   }

//   return (
//     <>
//       <Button
//         variant="ghost"
//         size="icon"
//         className={cn(
//           "fixed top-4 left-4 z-[60] md:hidden bg-sidebar text-sidebar-foreground hover:bg-sidebar-accent transition-all duration-200",
//           isMobileOpen && "left-[272px]", // Move button when sidebar is open
//         )}
//         onClick={toggleMobile}
//       >
//         {isMobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
//       </Button>

//       {isMobileOpen && <div className="fixed inset-0 bg-black/50 z-[45] md:hidden" onClick={toggleMobile} />}

//       <div
//         className={cn(
//           "fixed left-0 top-0 z-[50] h-screen bg-studio-bg border-r border-sidebar-border transition-all duration-300 ease-in-out",
//           "md:relative md:z-auto",
//           isCollapsed ? "w-16" : "w-64",
//           isMobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
//         )}
//         onMouseEnter={handleMouseEnter}
//         onMouseLeave={handleMouseLeave}
//       >
//         {/* Header */}
//         <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
//           {isCollapsed ? (
//             <Button
//               onClick={handleMenu}
//               variant="ghost"
//               className="w-full p-2 hover:cursor-pointer hover:bg-sidebar-accent flex items-center justify-center"
//             >
//               <Menu className="h-5 w-5 text-sidebar-foreground" />
//             </Button>
//           ) : (
//             <Button
//               onClick={handleMenu}
//               variant="ghost"
//               className="text-3xl font-bold hover:cursor-pointer dark:text-amber-100 hover:bg-transparent dark:hover:bg-transparent"
//             >
//               SimplX
//             </Button>
//           )}
//         </div>

//         {/* Content Area */}
//         <div className="flex flex-col">
//           {/* User Profile Section */}
//           <div className="flex p-4 border-b border-sidebar-border items-center justify-center">
//             <div className="flex items-center gap-3">
//               <Avatar className="h-10 w-10 flex-shrink-0">
//                 <AvatarImage src={session?.user.image || "/placeholder.svg"} alt={session?.user.name || "User Name"} />
//                 <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground">
//                   {session?.user.name
//                     ?.split(" ")
//                     .map((n) => n[0])
//                     .join("")
//                     .toUpperCase()}
//                 </AvatarFallback>
//               </Avatar>
//               {!isCollapsed && (
//                 <div className="flex-1 min-w-0">
//                   <p className="text-sm font-medium text-sidebar-foreground truncate">{session?.user.name}</p>
//                   <p className="truncate text-xs text-muted-foreground">{session?.user?.email || ""}</p>
//                 </div>
//               )}
//             </div>
//           </div>

//           {/* Navigation Area */}
//           <div className="flex-1 p-4">
//             {!isCollapsed && (
//               <div className="space-y-2">
//                 <div className="text-xs font-medium text-sidebar-foreground/70 uppercase tracking-wider">
//                   Navigation
//                 </div>
//                 <div className="space-y-1">
//                   <Button
//                     onClick={HandleHome}
//                     variant="ghost"
//                     className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
//                   >
//                     Home
//                   </Button>
//                   {/* <Button
//                     onClick={HandleDoucuments}
//                     variant="ghost"
//                     className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
//                   >
//                     Documents
//                   </Button> */}
//                   <Button
//                     variant="ghost"
//                     className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
//                   >
//                     Profile
//                   </Button>
//                 </div>
//               </div>
//             )}
//           </div>

//           {/* Sign Out Button */}
//           <div className="absolute bottom-0 p-4 border-t border-sidebar-border w-full flex flex-col justify-center items-center gap-3">
//             <ModeToggle isCollapsed={isCollapsed} />
//             <Button
//               variant="ghost"
//               onClick={handleSignOut}
//               className={cn(
//                 "text-destructive hover:bg-[#F4F4F4] hover:text-gray-700 transition-colors dark:hover:text-amber-100",
//                 isCollapsed ? "w-full p-2" : "w-full justify-start",
//               )}
//             >
//               <LogOut className="h-4 w-4" />
//               {!isCollapsed && <span className="ml-2">Sign Out</span>}
//             </Button>
//           </div>
//         </div>
//       </div>
//     </>
//   )
// }

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

// "use client"

// import { useState } from "react"
// import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
// import { Button } from "@/components/ui/button"
// import { Plus, Home, Globe, Building2, Bell, ArrowUpRight, Download, Menu, X } from "lucide-react"
// import { cn } from "@/lib/utils"
// import { signOut, useSession } from "next-auth/react"
// import { useRouter } from "next/navigation"
// import { useDispatch } from "react-redux"
// import { clearUser } from "@/store/slices/UserStoreSlice"
// import type { AppDispatch } from "@/store/store"
// import { modernToast } from "@/lib/toast"

// export function Sidebar() {
//   const router = useRouter()
//   const [isMobileOpen, setIsMobileOpen] = useState(false)
//   const { data: session } = useSession()
//   const dispatch = useDispatch<AppDispatch>()

//   const toggleMobile = () => setIsMobileOpen(!isMobileOpen)

//   const handleSignOut = () => {
//     modernToast.success(`Thanks for visiting, ${session?.user?.name || "User"}!`)
//     setTimeout(() => {
//       dispatch(clearUser())
//       localStorage.removeItem("user-storage")
//       localStorage.removeItem("app_toast_shown")
//       signOut({ callbackUrl: "/" })
//     }, 1200)
//   }

//   const handleHome = () => {
//     router.push("/")
//   }

//   const handleApplication = () => {
//     router.push("/application")
//   }

//   return (
//     <>
//       {/* Mobile toggle button */}
//       <Button
//         variant="ghost"
//         size="icon"
//         className={cn(
//           "fixed top-4 left-4 z-[60] md:hidden bg-gray-900 text-white hover:bg-gray-800 transition-all duration-200",
//           isMobileOpen && "left-[80px]",
//         )}
//         onClick={toggleMobile}
//       >
//         {isMobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
//       </Button>

//       {/* Mobile overlay */}
//       {isMobileOpen && <div className="fixed inset-0 bg-black/50 z-[45] md:hidden" onClick={toggleMobile} />}

//       {/* Sidebar */}
//       <div
//         className={cn(
//           "fixed left-0 top-0 z-[50] h-screen bg-gray-900 transition-all duration-300 ease-in-out w-20",
//           "md:relative md:z-auto",
//           isMobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
//         )}
//       >
//         <div className="flex flex-col h-full">
//           {/* Logo section */}
//           <div className="flex items-center justify-center p-4 border-b border-gray-800">
//             <div className="w-8 h-8 bg-white rounded-sm flex items-center justify-center">
//               {/* Perplexity-style geometric logo */}
//               <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gray-900">
//                 <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
//                 <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
//                 <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
//               </svg>
//             </div>
//           </div>

//           {/* Plus button */}
//           <div className="p-4">
//             <Button
//               variant="ghost"
//               size="icon"
//               className="w-full h-12 rounded-full border border-gray-700 hover:bg-gray-800 text-gray-300 hover:text-white transition-colors"
//               onClick={handleApplication}
//             >
//               <Plus className="h-5 w-5" />
//             </Button>
//           </div>

//           {/* Navigation items */}
//           <div className="flex-1 px-4 space-y-2">
//             <div className="space-y-1">
//               <Button
//                 variant="ghost"
//                 size="icon"
//                 className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors group"
//                 onClick={handleHome}
//               >
//                 <Home className="h-5 w-5 mb-1" />
//                 <span className="text-xs">Home</span>
//               </Button>

//               <Button
//                 variant="ghost"
//                 size="icon"
//                 className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors group"
//               >
//                 <Globe className="h-5 w-5 mb-1" />
//                 <span className="text-xs">Discover</span>
//               </Button>

//               <Button
//                 variant="ghost"
//                 size="icon"
//                 className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors group"
//               >
//                 <Building2 className="h-5 w-5 mb-1" />
//                 <span className="text-xs">Spaces</span>
//               </Button>
//             </div>
//           </div>

//           {/* Bottom section */}
//           <div className="px-4 pb-4 space-y-2 border-t border-gray-800 pt-4">
//             {/* Notification bell */}
//             <Button
//               variant="ghost"
//               size="icon"
//               className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
//             >
//               <Bell className="h-5 w-5" />
//             </Button>

//             {/* Account with pro badge */}
//             <div className="relative">
//               <Button
//                 variant="ghost"
//                 size="icon"
//                 className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
//               >
//                 <div className="relative">
//                   <Avatar className="h-6 w-6">
//                     <AvatarImage src={session?.user?.image || "/placeholder.svg"} alt={session?.user?.name || "User"} />
//                     <AvatarFallback className="bg-gray-700 text-white text-xs">
//                       {session?.user?.name
//                         ?.split(" ")
//                         .map((n) => n[0])
//                         .join("")
//                         .toUpperCase() || "U"}
//                     </AvatarFallback>
//                   </Avatar>
//                   {/* Pro badge */}
//                   <div className="absolute -bottom-1 -right-1 bg-cyan-500 text-white text-[10px] px-1 rounded-sm font-medium">
//                     pro
//                   </div>
//                 </div>
//                 <span className="text-xs mt-1">Account</span>
//               </Button>
//             </div>

//             {/* Upgrade */}
//             <Button
//               variant="ghost"
//               size="icon"
//               className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
//             >
//               <ArrowUpRight className="h-5 w-5 mb-1" />
//               <span className="text-xs">Upgrade</span>
//             </Button>

//             {/* Install */}
//             <Button
//               variant="ghost"
//               size="icon"
//               className="w-full h-12 flex flex-col items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
//               onClick={handleSignOut}
//             >
//               <Download className="h-5 w-5 mb-1" />
//               <span className="text-xs">Install</span>
//             </Button>
//           </div>
//         </div>
//       </div>
//     </>
//   )
// }
