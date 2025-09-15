import { configureStore } from '@reduxjs/toolkit';
import chatReducer from "./slices/ChatStoreSlice"
import userReducers from './slices/UserStoreSlice'
import fileReducer from './slices/fileSlice';

export const store = configureStore({
  reducer: {
    chat: chatReducer,  
    user: userReducers,
    files: fileReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;