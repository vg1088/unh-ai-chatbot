
# Known Issues

## Code Issues 
- **State Tracking:** The RAG does not switch which collection is being searched in runtime after it is initially assigned. The reset button has to be hit for the bot to change context

- **Repeat Answers:** The bot will occassionally repeat the last answer when it can't figure out what to do with the user input. "thank you" returns the correct message but "thanks" leads to a repeat message.

- ~~**UI Bugs:** If you click next to the text input box but not on the box, a cursor will appear in a "fake" intput box that doesn't let you type anything~~

## Future Improvements

**Streaming Responses**: The OpenAI API allows for response text to be streamed rather than delivered once finished, this would make the bot experience better for questions that require longer answers.

**Text Chunking**: The text is currently chunked with character splitting, a more sophisticated method like semantic chunking may provide better results.

**RAG as API function**: The OpenAI API allows you to add "tools" that can be called within your codebase, this might help if the bot needs to handle answers for a wider variety of situations

**Assistant API Endpoint**: Our bot currently uses the OpenAI ChatCompletion endpoint. The new assistants endpoint offers more features including streamlined state tracking for message histories.

**PDF Loading**: Having to load all the pdfs manually is complex for initial setup, we should probably be able to load them from an external list
