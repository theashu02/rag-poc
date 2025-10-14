"use client";

import { ReactNode, Suspense } from "react";
import { Provider as ReduxProvider } from "react-redux";
import { store } from "@/store/store";
import { ThemeProvider } from "@/components/ThemeFiles/theme-provider";
import SessionWrapping from "@/lib/Auth/sessionWrapping";
import { Toaster } from "@/components/ui/sonner";
import InitUser from "@/app/components/common/InitUser";
import Loading from "@/app/components/common/Loading";
import { SpacemanThemeProvider } from "@space-man/react-theme-animation";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<Loading />}>
      <ReduxProvider store={store}>
        <SessionWrapping>
          <SpacemanThemeProvider
            defaultTheme="system"
            defaultColorTheme="default"
            themes={["light", "dark", "system"]}
            colorThemes={["default", "supabase", "mono"]}
            // animationType="circle"
            duration={500}
          >
            <InitUser />
            {children}
            <Toaster />
          </SpacemanThemeProvider>
        </SessionWrapping>
      </ReduxProvider>
    </Suspense>
  );
}
