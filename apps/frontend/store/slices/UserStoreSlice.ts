'use client';

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UserState {
  userId: string | null;
  email: string | null;
  name: string | null;
  isAuthenticated: boolean;
}

const initialState: UserState = {
  userId: null,
  email: null,
  name: null,
  isAuthenticated: false,
};

export const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUser: (
      state,
      action: PayloadAction<{ userId: string; email: string; name: string }>
    ) => {
      state.userId = action.payload.userId;
      state.email = action.payload.email;
      state.name = action.payload.name;
      state.isAuthenticated = true;
    },
    clearUser: state => {
      state.userId = null;
      state.email = null;
      state.name = null;
      state.isAuthenticated = false;
    },
    updateAuthStatus: (state, action: PayloadAction<boolean>) => {
      state.isAuthenticated = action.payload;
    },
  },
});

export const { setUser, clearUser, updateAuthStatus } = userSlice.actions;
export default userSlice.reducer;