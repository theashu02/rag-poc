import { create } from "zustand";
import { persist } from "zustand/middleware";

interface UserState {
  userId: string | null;
  email: string | null;
  name: string | null;
  isAuthenticated: boolean;
  setUser: (user: { userId: string; email: string; name: string }) => void;
  clearUser: () => void;
  updateAuthStatus: (isAuthenticated: boolean) => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      userId: null,
      email: null,
      name: null,
      isAuthenticated: false,
      setUser: (user) =>
        set({
          userId: user.userId,
          email: user.email,
          name: user.name,
          isAuthenticated: true,
        }),
      clearUser: () =>
        set({
          userId: null,
          email: null,
          name: null,
          isAuthenticated: false,
        }),
      updateAuthStatus: (isAuthenticated) => set({ isAuthenticated }),
    }),
    {
      name: "user-storage",
      partialize: (state) => ({
        userId: state.userId,
        email: state.email,
        name: state.name,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
