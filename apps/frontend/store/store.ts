import { configureStore } from '@reduxjs/toolkit';
import chatReducer from "./slices/ChatStoreSlice"
import userReducers from './slices/UserStoreSlice'

export const store = configureStore({
  reducer: {
    chat: chatReducer,  
    user: userReducers,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;