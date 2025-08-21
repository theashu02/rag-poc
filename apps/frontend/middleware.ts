import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

export default withAuth(() => NextResponse.next(), {
  callbacks: { authorized: ({ token }) => !!token },
  pages: { signIn: "/" },
});

// protect any route that must be private
// export const config = {
//   matcher: ['/application/:path*'],
// };

// protect everything except “/”, API & static files
export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico|$).*)"],
};