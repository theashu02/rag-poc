"use client";

import { createSlice, PayloadAction, createAsyncThunk } from "@reduxjs/toolkit";
import api from "@/lib/ApiStore/axios";
import type { RootState } from "../store";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  isTyping?: boolean;
  isComplete?: boolean;
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  typingMessageId: string | null;
}

const generateId = () =>
  typeof window !== "undefined"
    ? window.crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export const sendMessage = createAsyncThunk<
  { messageId: string; content: string },
  string,
  { state: RootState; rejectValue: string }
>(
  "chat/sendMessage",
  async (content, { dispatch, rejectWithValue, getState }) => {
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content,
      timestamp: Date.now(),
      isComplete: true,
    };
    dispatch(addMessage(userMessage));

    const assistantMessage: Message = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isTyping: true,
      isComplete: false,
    };
    dispatch(addMessage(assistantMessage));
    dispatch(setTypingMessageId(assistantMessage.id));

    const userID = (getState() as RootState).user.userId;

    try {
      if (!userID) return rejectWithValue("User not authenticated");
      const res = await api.post("/api/v1/query/stream", { query: content, namespace: userID });
      const payload = res?.data;
      const answer =
        payload?.answer ??
        payload?.response ??
        payload?.message ??
        (typeof payload === "string" ? payload : JSON.stringify(payload)) ??
        "No response.";
      return { messageId: assistantMessage.id, content: answer };
    } catch (err: any) {
      return rejectWithValue(
        err?.response?.data?.message ?? err?.message ?? "Failed to send message"
      );
    }
  }
);

const initialState: ChatState = {
  messages: [],
  isLoading: false,
  error: null,
  typingMessageId: null,
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
    },
    updateMessageContent: (
      state,
      action: PayloadAction<{ id: string; content: string }>
    ) => {
      const msg = state.messages.find((m) => m.id === action.payload.id);
      if (msg) msg.content = action.payload.content;
    },
    setMessageComplete: (state, action: PayloadAction<string>) => {
      const msg = state.messages.find((m) => m.id === action.payload);
      if (msg) {
        msg.isTyping = false;
        msg.isComplete = true;
      }
    },
    setTypingMessageId: (state, action: PayloadAction<string | null>) => {
      state.typingMessageId = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    clearMessages: (state) => {
      state.messages = [];
      state.typingMessageId = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(sendMessage.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(sendMessage.fulfilled, (state, action) => {
        state.isLoading = false;
        state.messages = state.messages.map((msg) =>
          msg.id === action.payload.messageId
            ? {
              ...msg,
              content: action.payload.content,
              isTyping: true,
              isComplete: false,
            }
            : msg
        );
        state.typingMessageId = action.payload.messageId;
      })
      .addCase(sendMessage.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload ?? "Failed to send message";
        /* replace assistant placeholder with error text */
        state.messages = state.messages.map((msg) =>
          msg.id === state.typingMessageId
            ? {
              ...msg,
              content: "Sorry, I encountered an error. Please try again.",
              isTyping: false,
              isComplete: true,
            }
            : msg
        );
        state.typingMessageId = null;
      });
  },
});

export const {
  addMessage,
  updateMessageContent,
  setMessageComplete,
  setTypingMessageId,
  clearError,
  clearMessages,
} = chatSlice.actions;

export default chatSlice.reducer;
